import asyncio, os
from mppi.comm.ws_client_sync import ClientConfig, infer_once
cfg = ClientConfig(url="ws://127.0.0.1:9010", request_timeout_s=30.0)
resp = asyncio.run(infer_once(cfg, q=[0.0]*7, gripper=0.0, step_id=0))
print("ON", resp["payload"]["server_timing"])