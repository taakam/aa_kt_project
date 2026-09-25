from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

config = SO101FollowerConfig(
    port="COM6",
    id="my_awesome_follower_arm",
)
follower = SO101Follower(config)
follower.setup_motors()

