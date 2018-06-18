# AI_2048
In this project, I use adversarial search to play the 2048. You can play it by yourself on https://gabrielecirulli.github.io/2048/.

There are two agents existing in this game, Computer AI and Player AI. The former generates a new tile, either a 2(90%) or a 4(10%), while the Player AI decides what move it is going to make. The Player AI will play as if the computer is adversarial since this proves more effective in beating the game.
Since the process of generating a tile is concerned with probability, the expectiminimax algorithm is employed here.
With expectiminimax, the game playing strategy assumes the Computer AI chooses a tile to place in a way that minimizes the Player's outcome.
The Chance step and Minimize step are merged together to make the code look neat.

