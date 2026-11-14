"""Causality, cached-step alignment and legacy compatibility of public history."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import h5py
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install optional requirements-learning.txt")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.public_history import PublicHistory, HISTORY_DIM, normalize_history, reconstruct_history
from learning.diffusion_policy import ActionDiffusion, ModelConfig, Tokenizer, DiffusionPolicy, CHECKPOINT_SCHEMA
from learning.data import CompactChunkDataset, load_public_history
from scripts.run_research import TimedPolicy
from scripts import train_policy
from utils.policy import PolicySession

CONTRACT = dict(id="forte-joint-reference-v2", shape=[8], arm_joint_count=7,
                policy_hz=20., joint_reference_speed_rad_s=.25,
                joint_lower_limits_rad=[-2.] * 7, joint_upper_limits_rad=[2.] * 7,
                gripper_closed_m=0., gripper_open_m=.074, gripper_speed_limit_m_s=.1,
                cameras=["front", "wrist"], image_size_hw=[24,32], privileged_fields_are_policy_inputs=False)


def observation(proprio):
    return dict(proprio=np.asarray(proprio,np.float32), instruction="pick red cube into blue tray",
                rgb={camera:np.zeros((24,32,3),np.uint8) for camera in CONTRACT['cameras']})


def episode(path, count=12):
    props=np.zeros((count+1,23),np.float32)
    props[:,0]=np.arange(count+1)*.001
    props[:,14]=.05
    commands=np.tile(np.array([.4,0,0,0,0,0,0,-1],np.float32),(count,1))
    with h5py.File(path,'w') as f:
        f.attrs['action_contract']=json.dumps(CONTRACT)
        f.create_dataset('observations/proprio',data=props)
        f.create_dataset('transitions/proposed_action',data=commands)
        f.create_dataset('transitions/action',data=-commands)  # Hidden override must not become command history.
        f.create_dataset('privileged/poison',data=np.nan)
    return props,commands


class PublicHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model_policy(self,steps=4):
        config=ModelConfig(image_height=24,image_width=32,hidden_dim=32,diffusion_steps=8,
                           inference_steps=4,state_history_steps=steps,gripper_mode='binary')
        tokenizer=Tokenizer.fit(['pick red cube into blue tray'])
        model=ActionDiffusion(config,len(tokenizer.words))
        policy=DiffusionPolicy(model,tokenizer,dict(mean=[.1]*23,std=[2.]*23),CONTRACT,{},'cpu')
        return model,policy

    def test_configuration_and_legacy_parameter_initialization(self):
        for invalid in (-1,33,1.5,True,None):
            with self.assertRaises(ValueError): ModelConfig(state_history_steps=invalid)
        old=asdict(ModelConfig());old.pop('state_history_steps')
        self.assertEqual(ModelConfig(**old).state_history_steps,0)
        torch.manual_seed(72);old_model=ActionDiffusion(ModelConfig(hidden_dim=32),12)
        torch.manual_seed(72);new_model=ActionDiffusion(ModelConfig(hidden_dim=32,state_history_steps=8),12)
        for key,tensor in old_model.state_dict().items():
            torch.testing.assert_close(tensor,new_model.state_dict()[key],rtol=0,atol=0)
        self.assertTrue(any(key.startswith('history_encoder.') for key in new_model.state_dict()))

    def test_reference_and_jaw_reconstruction_padding_and_units(self):
        history=PublicHistory(CONTRACT,3)
        p=np.zeros(23,np.float32);p[14]=.05
        history.observe_proprio(p)
        raw=history.raw()
        np.testing.assert_array_equal(raw[:2],0)
        self.assertEqual(raw[-1,47],1);self.assertEqual(raw[-1,48],0)
        history.observe_action(np.array([1]*7+[-1],np.float32))
        history.observe_proprio(p)
        row=history.raw()[-1]
        np.testing.assert_allclose(row[31:38],.00625)
        np.testing.assert_allclose(row[39:46],.0125)
        self.assertAlmostEqual(float(row[38]),2*(.05-.005)/.074-1,places=6)
        self.assertAlmostEqual(float(row[46]),-.005/.074,places=6)
        np.testing.assert_array_equal(row[23:31],[1]*7+[-1]);self.assertEqual(row[48],1)
        normalized=normalize_history(history.raw(),np.ones(23),np.ones(23)*2)
        np.testing.assert_array_equal(normalized[0],0)
        np.testing.assert_allclose(normalized[-1,:23],(p-1)/2)
        np.testing.assert_array_equal(normalized[-1,23:],row[23:])

    def test_future_observation_and_action_changes_cannot_change_current_features(self):
        props=np.zeros((13,23),np.float32);commands=np.zeros((12,8),np.float32)
        original=reconstruct_history(props,commands,CONTRACT,4)
        changed_props=props.copy();changed_props[6:]=.3
        changed_commands=commands.copy();changed_commands[5:]=1
        changed=reconstruct_history(changed_props,changed_commands,CONTRACT,4)
        np.testing.assert_array_equal(original[:6],changed[:6])
        self.assertFalse(np.array_equal(original[6],changed[6]))

    def test_missing_duplicate_feedback_and_invalid_inputs_fail(self):
        h=PublicHistory(CONTRACT,4)
        with self.assertRaises(RuntimeError):h.raw()
        with self.assertRaises(RuntimeError):h.observe_action(np.zeros(8))
        with self.assertRaises(ValueError):h.observe_proprio(np.zeros(24))
        h.observe_proprio(np.zeros(23))
        with self.assertRaises(RuntimeError):h.observe_proprio(np.zeros(23))
        with self.assertRaises(ValueError):h.observe_action(np.ones(8)*2)
        h.observe_action(np.zeros(8))
        with self.assertRaises(RuntimeError):h.observe_action(np.zeros(8))
        with self.assertRaises(ValueError):reconstruct_history(np.zeros((3,23)),np.zeros((3,8)),CONTRACT,4)

    def test_loader_uses_issued_commands_without_privileged_or_executed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.h5';props,commands=episode(path)
            expected=reconstruct_history(props,commands,CONTRACT,4)
            original=h5py.Group.__getitem__
            def guarded(group,key):
                if str(key).startswith('privileged'):raise AssertionError('Privileged read')
                return original(group,key)
            with patch.object(h5py.Group,'__getitem__',guarded):
                np.testing.assert_array_equal(load_public_history(path,CONTRACT,4),expected)
            with h5py.File(path,'a') as f:del f['transitions/proposed_action']
            with self.assertRaises(ValueError):load_public_history(path,CONTRACT,4)

    def test_dataset_uses_physical_indices_through_unlabeled_gaps_and_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.h5';props,commands=episode(path)
            config=ModelConfig(image_height=24,image_width=32,state_history_steps=4)
            tokenizer=Tokenizer.fit(['pick red cube into blue tray'])
            records=[dict(path=str(path),transitions=12,supervised_transitions=3,action_contract=CONTRACT)]
            samples=[dict(observation=observation(props[t]),action=np.zeros((8,8),np.float32),
                          action_mask=np.ones(8,bool),transition_index=t,is_intervention=False) for t in (0,3,5)]
            with patch('learning.data.training_chunks',return_value=iter(samples)):
                data=CompactChunkDataset(records,config,tokenizer,split='train',anchor_stride=2)
            expected=reconstruct_history(props,commands,CONTRACT,4)
            np.testing.assert_array_equal(data.history[:len(data)],expected[[0,5]])
            data.set_normalization(data.normalization())
            self.assertEqual(len(data[1]),6)
            np.testing.assert_array_equal(data[1][-1].numpy(),normalize_history(expected[5],data.mean,data.std))

    def test_policy_session_matches_offline_history_at_cached_chunk_decisions(self):
        model,policy=self.model_policy()
        issued=np.tile(np.array([.4,0,0,0,0,0,0,-1],np.float32),(12,1))
        props=np.zeros((13,23),np.float32);props[:,0]=np.arange(13)*.001;props[:,14]=.05
        expected=reconstruct_history(props,issued,CONTRACT,4)
        seen=[]
        def predict(images,proprio,tokens,*,generator,history):
            seen.append(history.cpu().numpy()[0].copy())
            return torch.from_numpy(np.tile(issued[0],(1,8,1)))
        session=PolicySession(TimedPolicy(policy),gym.spaces.Box(-1.,1.,(8,),dtype=np.float32),execute_actions=3)
        session.reset(seed=0)
        with patch.object(model,'sample',side_effect=predict):
            for t in range(12):np.testing.assert_array_equal(session(observation(props[t])),issued[t])
        self.assertEqual(len(seen),4)
        for actual,index in zip(seen,(0,3,6,9)):
            np.testing.assert_array_equal(actual,normalize_history(expected[index],policy.history_mean,policy.history_std))
        np.testing.assert_allclose(policy.history.codec.arm_reference[0],.06,atol=1e-8)
        # run_research resets the actual policy, then clears the timed session.
        policy.reset(seed=0)
        session.reset(seed=0)
        self.assertIsNone(policy.history.last_proprio)
        with patch.object(model,'sample',side_effect=predict):session(observation(props[0]))
        np.testing.assert_array_equal(seen[-1],seen[0])

    def test_callback_receives_a_copy_before_issued_action(self):
        events=[]
        class Policy:
            def observe_proprio(self,p):events.append('observe');p[:]=100
            def predict(self,obs):
                np.testing.assert_array_equal(obs['proprio'],0);events.append('predict')
                return np.zeros((2,8),np.float32)
            def observe_action(self,a):events.append('action')
        session=PolicySession(Policy(),gym.spaces.Box(-1.,1.,(8,),dtype=np.float32),execute_actions=2)
        session(observation(np.zeros(23)));session(observation(np.zeros(23)))
        self.assertEqual(events,['observe','predict','action','observe','action'])

    def test_history_policy_rejects_direct_calls_without_both_feedbacks(self):
        model,policy=self.model_policy();obs=observation(np.zeros(23))
        with self.assertRaisesRegex(RuntimeError,'observation feedback'):policy.predict(obs)
        policy.observe_proprio(obs['proprio'])
        with patch.object(model,'sample',return_value=torch.zeros(1,8,8)):
            policy.predict(obs)
            with self.assertRaisesRegex(RuntimeError,'action feedback'):policy.predict(obs)
            policy.observe_action(np.zeros(8))
            with self.assertRaisesRegex(RuntimeError,'observation feedback'):policy.predict(obs)

    def test_legacy_checkpoint_without_history_field_remains_exact(self):
        model,policy=self.model_policy(0);obs=observation(np.zeros(23))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.pt';cfg=asdict(model.config);cfg.pop('state_history_steps')
            torch.save(dict(schema=CHECKPOINT_SCHEMA,model_config=cfg,vocabulary=policy.tokenizer.words,
                            normalization=dict(mean=[.1]*23,std=[2.]*23),action_contract=CONTRACT,
                            metadata={},step=1,model_state=model.state_dict()),path)
            restored=DiffusionPolicy.load(path,CONTRACT,device='cpu')
            policy.reset(seed=17);restored.reset(seed=17)
            np.testing.assert_array_equal(policy.predict(obs),restored.predict(obs))
            self.assertIsNone(restored.history)

    def test_history_forward_backward_and_missing_history_fail_closed(self):
        model,_=self.model_policy()
        images=torch.zeros(2,2,3,24,32,dtype=torch.uint8);proprio=torch.zeros(2,23)
        tokens=torch.ones(2,64,dtype=torch.int64);actions=torch.zeros(2,8,8);mask=torch.ones(2,8,dtype=torch.bool)
        with self.assertRaises(ValueError):model.training_loss(images,proprio,tokens,actions,mask)
        history=torch.randn(2,4,HISTORY_DIM)
        loss=model.training_loss(images,proprio,tokens,actions,mask,history)
        loss.backward()
        for parameter in (model.history_encoder[0].weight,model.output.weight,model.word_embedding.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all());self.assertGreater(parameter.grad.abs().sum().item(),0)

    def test_validation_weights_action_masks_not_history_tensor(self):
        class Model:
            config=ModelConfig(state_history_steps=4)
            def eval(self):return self
            def training_loss(self,*args,generator):return torch.tensor(float(len(args[0])))
        def batch(n,valid):
            return [torch.zeros(n,2,3,24,32),torch.zeros(n,23),torch.ones(n,64,dtype=torch.long),
                    torch.zeros(n,8,8),torch.tensor([[True]*valid+[False]*(8-valid)]*n),
                    torch.ones(n,4,HISTORY_DIM)*-99]
        result=train_policy.validation_loss(Model(),[batch(2,8),batch(1,4)],torch.device('cpu'),0)
        self.assertAlmostEqual(result,(2*16+1*4)/20)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA optional')
    def test_history_model_deterministic_cuda_backward(self):
        previous=torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            model,_=self.model_policy();model=model.to('cuda').train()
            loss=model.training_loss(torch.zeros(2,2,3,24,32,dtype=torch.uint8,device='cuda'),
                torch.zeros(2,23,device='cuda'),torch.ones(2,64,dtype=torch.long,device='cuda'),
                torch.zeros(2,8,8,device='cuda'),torch.ones(2,8,dtype=torch.bool,device='cuda'),
                torch.ones(2,4,HISTORY_DIM,device='cuda'))
            loss.backward();self.assertTrue(torch.isfinite(loss))
        finally:torch.use_deterministic_algorithms(previous)

    def test_cli_one_update_saves_history_semantics_and_validates_checkpoint(self):
        from test_text_pooling import TinyDataset, fixture_record
        class HistoryDataset(TinyDataset):
            def __getitem__(self,index):
                return super().__getitem__(index)+(torch.ones(4,HISTORY_DIM),)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'fit'
            argv=['train_policy.py','--train','fixture.h5','--output',str(output),'--device','cpu',
                  '--steps','1','--batch-size','4','--threads','1','--image-size','24','32',
                  '--hidden-dim','32','--diffusion-steps','8','--inference-steps','4','--state-history-steps','4']
            old=torch.are_deterministic_algorithms_enabled()
            try:
                with patch.object(sys,'argv',argv),patch.object(train_policy,'expand_paths',return_value=['fixture.h5']), \
                        patch.object(train_policy,'inspect_episodes',return_value=([fixture_record()],[])), \
                        patch.object(train_policy,'CompactChunkDataset',side_effect=HistoryDataset):
                    train_policy.main()
            finally:torch.use_deterministic_algorithms(old)
            saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            self.assertEqual(saved['model_config']['state_history_steps'],4)
            self.assertEqual(saved['metadata']['preprocessing']['rgb_history_steps'],1)
            self.assertEqual(saved['metadata']['public_history']['steps'],4)
            self.assertEqual(saved['metadata']['initialization']['kind'],'from_scratch')


if __name__=='__main__':unittest.main()
