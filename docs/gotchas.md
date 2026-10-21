# Gotchas (one line each; the code site carries a GOTCHA(G-###) comment)
G-001 lerobot/diffusion_pusht on the Hub is in LeRobot's pre-processor format (stats inside the weights); plens migrates it once into .cache/migrated/.
G-002 Forward hooks on nn.ModuleList/ModuleDict never fire (their forward never runs). Hook a child; the recorder refuses containers.
G-003 gym-pusht returns 96x96 images regardless of LeRobot's 384 default, and the policy expects 96. Pinned in envs.py.
G-004 Vector-env same-step autoreset: on the last step `info` is already the next episode's; real data is in info["final_info"]. Else success and coverage of the final step are wrong.
G-005 LeRobot's migration CLI crashes saving configs whose tuples came back from JSON as lists (crop_shape); the Hub config also says device "cuda". plens migrates in-process and fixes both.
G-006 diffusion_pusht runs 100 denoising passes per re-plan (every 8 steps): ~10 min per 300-step episode on a CPU. Start with --n 10 on a Mac.
