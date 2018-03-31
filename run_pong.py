from agents import PongAgent


if __name__ == '__main__':
    pong_agent = PongAgent()
    pong_agent.train(n_iters=500, p_steps=20)
