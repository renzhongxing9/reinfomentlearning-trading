# reinfomentlearning-trading
本项目采用深度强化学习方法构建了一个以沪深300指数为标的的量化择时策略。采用DQN算法和基于OpenAI的Gym框架搭建智能体与交易环境的行为交互。经训练样本外始超参数测试集年化收益率达到5.34%，夏普比率0.5。

项目文件夹Reinforcement_learning_trading下共包含6个代码文件。其中前五个文件是强化学习所需的智能体组件，在这里已经全部调试好并在.py文件中封装，运行项目代码是无需改动。run.ipynb文件是本项目的调用文件，在该文件中完成对上述代码的调用和全局超参数的修改

├─ Agent.py

├─ Environment.py

├─ Functions.py

├─ Memory.py

├─ Model.py

└─ run.ipynb
