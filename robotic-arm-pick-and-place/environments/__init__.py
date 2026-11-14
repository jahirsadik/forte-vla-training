"""Task scene loading independent of the legacy cube controllers."""

"""Import this package to register the custom Forte Gymnasium benchmarks."""

from gymnasium.envs.registration import register

register(id="ForteBookshelf-v3", entry_point="environments.bookshelf_task:BookshelfEnv")
register(id="ForteSorting-v3", entry_point="environments.sorting:SortingEnv")
register(id="ForteBookshelfHard-v1", entry_point="environments.bookshelf_hard:BookshelfHardEnv")
