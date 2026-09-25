from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

config = SO101FollowerConfig(
    port="COM3",
    id="my_awesome_follower_arm",
)

follower = SO101Follower(config)
follower.connect(calibrate=False)
follower.calibrate()
follower.disconnect()