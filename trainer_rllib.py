from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution import TrainOneStep, UpdateTargetNetwork, Concurrently, StandardMetricsReporting,\
                                Replay, StoreToReplayBuffer, ParallelRollouts
from ray.rllib.execution.replay_buffer import LocalReplayBuffer


from policy_rllib import DQNRLlibPolicy
from config import *


def execution_plan(workers, config):
    local_replay_buffer = LocalReplayBuffer(
        num_shards=1,
        learning_starts=config['learning_starts'],
        buffer_size=config['buffer_size'],
        replay_batch_size=config['train_batch_size'],
    )

    store_op = ParallelRollouts(workers, mode='bulk_sync') \
        .for_each(StoreToReplayBuffer(local_buffer=local_replay_buffer))

    replay_op = Replay(local_buffer=local_replay_buffer) \
        .for_each(TrainOneStep(workers)) \
        .for_each(UpdateTargetNetwork(workers, config['target_network_update_freq']))

    train_op = Concurrently([store_op, replay_op],
                            mode='round_robin',
                            output_indexes=[1])

    return StandardMetricsReporting(train_op, workers, config)


DQNRLlibTrainer = build_trainer(
    name='DQNRLlibTrainer',
    default_policy=DQNRLlibPolicy,
    default_config=RLLIB_DEFAULT_CONFIG,
    execution_plan=execution_plan
)
