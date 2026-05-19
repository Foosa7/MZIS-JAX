from celery import Celery
import time
import math
import json
import redis

# Initialize Celery app with Redis broker
app = Celery('mzix_worker', broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')

# Optional: configure celery to support task priorities (Redis usually supports 0-9)
app.conf.broker_transport_options = {
    'priority_steps': list(range(10)),
    'queue_order_strategy': 'priority',
}

# Redis client for pub/sub telemetry streaming
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.task(bind=True)
def process_batch_job(self, payload_dict: dict):
    job_id = payload_dict['job_id']
    user_id = payload_dict['user_id']
    unitaries = payload_dict['unitaries']
    
    print(f"Starting job {job_id} for user {user_id} with {len(unitaries)} unitaries")
    
    for idx, target in enumerate(unitaries):
        name = target['name']
        print(f"Processing unitary: {name}")
        
        # 1. Warm Start Translation (Current -> Phase)
        currents = target['initial_currents_ma']
        init_phases = {}
        for mzi, vals in currents.items():
            # Titanium Nitride Non-linear thermal model constants
            c_res = 1.04
            alpha_res = 0.0034
            p_2pi = 24.5
            
            # theta mapping
            i_th = vals['heater_theta']
            r_th = c_res * (1 + alpha_res * (i_th ** 2))
            p_th = (i_th ** 2) * r_th
            phase_th = p_th * (2 * math.pi / p_2pi) # + gain/bias per heater
            
            # phi mapping
            i_phi = vals['heater_phi']
            r_phi = c_res * (1 + alpha_res * (i_phi ** 2))
            p_phi = (i_phi ** 2) * r_phi
            phase_phi = p_phi * (2 * math.pi / p_2pi) # + gain/bias per heater
            
            init_phases[mzi] = {'theta': phase_th, 'phi': phase_phi}
            
        # 2. JAX Optimization (Hardware-in-the-Loop)
        # Mocking the gradient descent steps
        for step in range(1, 11):
            time.sleep(0.5) # Simulate hardware evaluation & JAX forward/backward pass
            
            loss = 1.0 / (step * 0.5 + 1.0) # mock loss decreasing
            
            # Send telemetry to FastAPI via Redis PubSub
            telemetry_data = {
                "job_id": job_id,
                "unitary": name,
                "iteration": step,
                "loss": loss
            }
            redis_client.publish(f"telemetry:{user_id}", json.dumps(telemetry_data))
            print(f"  Step {step}/10 for {name}: Loss = {loss:.4f}")
            
        # 3. Phase -> Current (Hardware Push)
        print(f"Finished optimizing {name}")
        
    return {"status": "completed", "job_id": job_id}
