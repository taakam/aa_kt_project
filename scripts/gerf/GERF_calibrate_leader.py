from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

config = SO101LeaderConfig(
    port="COM4",
    id="my_awesome_leader_arm",
)
leader = SO101Leader(config)
leader.setup_motors()