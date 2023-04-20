# Gloom: Modular Reinforcement Learning for Fighting Game AI #

![Gloom banner image](banner.png)

Gloom is a modular pipeline for training deep Q-learning agents to reporduce complex, human-like strategies in fighting games.

Gloom is used to train a multi-stage deep Q-learning pipeline which will use different models for the different stages during a match in a fighting game. By implementing different models for different stages of a match we propose that different sub-goals can be accurately and fluidly defined for the AI player during these different stages. We have defined a model for both the neutral and turns stages - the neutral model - and a separate model for the okizeme stage - the okizeme model. The neutral model allows us to craft a specific reward function that motivates the model into taking actions that will result in knocking down their opponent. Then, the okizeme model can be used to predict the opponents movements once they have been knocked down and devise a plan that results in the opponent taking maximal damage once they stand back up. The okizeme model was designed to be allowed to perform actions for 90 frames, where $1 frame = 0.0167s$.

This repository utilizes the [fightingICE environment](https://github.com/TeamFightingICE/Gym-FightingICE). More information is available at the [official FightingICE website](http://www.ice.ci.ritsumei.ac.jp/~ftgaic/).

Banner generated from [this website](https://liyasthomas.github.io/banner/). Art taken from [UNDER NIGHT IN-BIRTH Exe:Late[cl-r]](https://www.blazblue.jp/tag/manual-switch/en/character/uni.html)

## TODO ##

----- DISTRIBUTIONAL Q NETWORK -----

----- NOISY NETS -----

----- SOFT Q-LEARNING -----