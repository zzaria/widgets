---
author: aaeria
---

Here's a fireworks simulation. It's a nice decoration for holidays and syncs up with whatever music you are listening to. You can set it as your desktop wallpaper using [Lively](https://github.com/rocksdanister/lively).

[https://github.com/zzaria/fireworks](https://github.com/zzaria/fireworks)

<video controls><source src="{{'/assets/images/2024-02-14 fireworks.webm' | relative_url}}"></video>

## Behind the scenes

The program times the fireworks based on when the base (volume below a certain frequency) suddenly increases. However, having too many fireworks going off all the time would be a mess, so I decided to aim for at most 2.5 firework events per second, prioritizing the heaviest beats (largest volume spikes). To do this the program checks if the current spike is within the top 5 largest in the previous 2 seconds. Then, to add more activity during exciting parts of the song, the program checks the overall volume. If it is above a threshold, the explosion size and target fire rate are increased and there is a chance to fire multiple fireworks in a single event.