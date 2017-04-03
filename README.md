# Team

# Issues

1. Takes a lot of time due to inconsequential actions which lead to infinte loops (if not exploring properly). Example would be a predicted downward action when no cell can move any downwards.
   * Explore more
   * Increase batch size for gradient descent update to avoid very local minimas.

2. The implementation with target Network is taking a lot of time!
  
3. Work to improve code readability

4. Play with Hyperparams:
   * N: Total number of experiences in the replay buffer
   * B: Number of experiences sampled from replay

5. Beyond this project - Additional Deep Q learning tweaks that may improve results. We encourage experimenting with following:
   * Double DQN
   * Dueling Network
   * Prioritized Replay
