---
layout: page
title: "Clash-Royale AI agent ⚔️"
---

## Online Agent Visualization 🎮🤖

<img src="output.gif" alt="AI Agent demo" width="1000" style="border-radius: 10px;">

- This is fully controlled by the policy network that I use scrcpy to let the mouse to control the android phone to select the card and place on the arena. (Make sure to turn on the debug mode of android phone).
## 🔬 Brief Project Flow 📝

- Since we have no access to the internal control for Clash Royale, so that no direct information to extract state, action, and reward for traditional online training in Reinforcement Learning. We have to build the visual detection block to extract the necessary information for training data. 

- Refer to the [GitHub repo](https://github.com/shawnxu0407/Clash_Royale_agent) for the code.

1. **Dataset/Episode Collection 📸**
   - 1.1 Use [Scrcpy](https://github.com/Genymobile/scrcpy) and **FFmpeg** to access the Android phone screen and extract the game episode frames (later add the link for the dataset collection). Notice we have to use the Expert datasets.
   
   - 1.2 Prepare troops segments for generative arena images, which is the training dataset for visual detection model YOLOv8:
  
   ![Generative Arena View](generative_arena.png)
     
3. **Visual Detection/ Data Processing 🔍**
   - 2.1 **State builder** (torch_state_builder.py): Use trained YOLOv8 for arena information extraction to build state **s**, use ResNet classification to extract info for deployed cards and elixir, use cnocr to extract actual time.
   
   - 2.2 **Action builder** (torch_action_builder.py): Use ResNet classification to visually detect card deployments.
   
   - 2.3 **Reward builder** (torch_reward_builder.py): Use cnocr to detect queen and king tower HP deduction.
   
   - 2.4 Every frame would have the corresponding state (s) 🟢, action (a) ⚡, and reward (r) ⭐

4. **Policy Network Construction 🧠**
   
 ![Policy Network Preview](policy_model_en.png)

6. **AWS Sagemaker Training ☁️**
   - I used AWS sagemaker to train, first to make sure the combination of different models work well on the local environment using train.py script, then put the data into the AWS S3 bucket for later data loading. The core is to create a train_sagemaker.py to wrap the original training script.
   - Basically, AWS sagemaker would use a Linux env to create a docker image which are defined in the requirements.txt. It also uploads all the related script in the root folder. Then if the local test works well, the clould training should be working as well. :)

## Training Dataset Construction 🗂️

1. The keys of **state (s)** are [arena, cards, elixir]. The trained YOLO model gives inference on arena detection results and detected troops with keys [position, class, belongings]. The ResNet model detects the cards table, showing only 4 cards at a time. cnocr detects remaining elixir.

2. The keys of **action (a)** are [select, pos] representing selected cards and deployment positions in the arena. The arena is divided into 32x18 sub-blocks.

3. The key of **reward (r)** is the scalar reward 🏅. Positive reward if opposite tower HP is reduced, negative otherwise.

4. Dataloading process: each timestep t has (s, a, r) pairs. The sequential construction is necessary for transformer decoder model.
   - Sequential Dataloader: Each sequence is length n (ex. 30), with the first 29 input pairs predicting the 30th pair.

5. **Sparse Action Addressing ⚡**
   
 ![Frame Re-weighting](frame_weights.png)

The actual action frames get higher weights 🏋️‍♂️, and neighboring frames get decaying weights.

## Model Structure Discussion 🏗️

0. Quick View of Model
   - View the whole model as a learning framework that takes the first $n-1$ items and produce the $n$-th item like a time sequence modeling where we can "open" each item spatially as we have lots of information for each time on the game arena like deployed cards, time, what the arena looks like (any enemy on the playground?)...



1. Spatial Latent representation
   - The very first important concept in this model uses the cross attention mechanism to produce the spatial latent representation at each time step $l_t$.
  


   - For each fixed time step t, apply a cross attention mechanism on [state_embedding, action_embedding, reward_embedding], where state_embedding is [image_patches_embedding, elixir_embedding, cards_embedding], action_embedding is [select_embedding, position_embedding] and reward_embedding is purely the embedding of scalr reward of each time step.

  
   
   - The reason to apply cross attention to the combination of embedding is try to build the similarity scores between items. For example, the image patches sub-divide the original image into smaller parts, and use cross attention, the model would learn that the cards deployment position is more related to one of all patches that some specific cards is more related to some specific part of whole arena which is related to some high rewards for fixed time.
  


   - The purpose of whole spatial latent representation is trying to learn a high dimensional representation of vectors to represent the "similarities among all items spatially" for each time step.



2. Time Sequence Modeling
   - Given we have the spatial local latent representation $l_t$, we also build the global representations $x^t_g$. In this case, the emphasis is no longer on the local spatial (arena) side but on the reasoning over time steps.
  

   - We first embed the image and cards, and concatenate spatial latent $l_t$ to form $x^t_g$ at time t, and we apply cause attention to all the global embeddings $x^t_g$.

  
   - The cause attention works like this: when we work on the global embedding at time step $t=9$, we compute the similarties score of $x^9_g$ among all the previous embeddings. We do not compute the future similarities as we try to let the model make decision on the previous steps without knowing any future information. (otherwise it is cheating!)

## Future Work and Summary
- My model performance is actually really bad. My guess is that I did not collect enough games/episodes for the training.


- Each episode contains around 900 frames, and not all the frames will be into the dataset depending on the selected weights.


- If you would like to reproduce the model for your own card selection, you have to re-train the policy network from scratch as the card information will be different, I have not fine tunned yet.


- As you can notice that this model only validate for a specific card set, it is not valid for the all the cards. Maybe training a policy network for all the card would be interesting but the dataset size it require will be very very expensive.


- Treat training dataset is a way to represent the knowledge domain where this domain is somehow crazily high dimensional. Given this, we have to provide the data to the model that what card I select and the position I place this card for all possible siuations. The dataset will be intense so that it make the knowledge domain being smooth enough for model learning. (Emmm this is just my personal/shadow thinking).


- This is offline learning, I can not access the live interaction with the game internal so the tradional RL can not be applied. If the game interanl can be accessed, then we do not need any visual detection to extract information and RL algorithm and improve the model though interacting within games.


## References 📚
1. [Clash Royale AI studies](https://wty-yy.xyz/posts/12073/)

2. [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)

