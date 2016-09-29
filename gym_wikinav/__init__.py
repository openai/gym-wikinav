from gym.envs.registration import register


register(
    id="wikinav-v0",
    entry_point="gym_wikinav.envs:EmbeddingWikiNavEnv",
    timestep_limit=50)
