# Project Syna
 
Syna aims to be a Vtuber with distinct personality

it will use llama3.2-8b as the base model (improve once sustem imrpoves)

tech stack:
- python - AI model
- C# and python - game interaction
- nextjs/ts - front-end control panel
- mySql - memory database?


we will train it to be
- Entertaining
- Emo
- Mumei?
- likes rock and metal music
- Likes to sing and play guitar

MVP for v0.1 alpha
- use custom trained model
- bot is able to read and reply chat messages
- bot is able to join voice chat (Todo: set up PyNaCl so voice will be supported)

MVP for v0.2 alpha
- bot is able to hear and reply
- connect bot with control panel
- VTuber model/ assets

MVP for v0.3 alpha
- Understand twitch streams
    - chat
    - donations
    - raids and etc


future planning
- play games (minecraft, hollow knight, etc)
- sing and play "guitar"
- collab streams
- more functionalities with discord

Roadmap
1. Gather datasets
    - Synthetic data
    - Huggingface
    - Reddit
        - r/jokes
        - r/funny
        - r/sarcasm
        - r/mumei
    - Twitter
    - SARC (Sarcasm Corpus)
    - Discord chatlogs

1.2. format datasets
{
    "conversation": [
        "role": role,
        "message": text,
        "tone": tone
    ]
}

1.3. train order
- entertaining
- Conversational and reasoning

2. Train the model
Setup:
CPU: Ryzen 5800x3D
GPU: RTX 4070 ti

- QLoRA allow for lighter loads

3. Deploy model to pycord
4. Text To Speech (whisper) - output
    - Allow bot to join voice calls
    - Allow bot to speak on stream
5. Speech To Text (?) - input
    - Allow bot to listen and form respond

6. profit???

