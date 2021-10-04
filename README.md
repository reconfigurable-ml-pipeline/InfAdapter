*Horizontal Autoscaling of Web Applications in Distributed Systems*
-

>### Implementation of an auto-scaling recommender for containerized web applications in kubernetes cluster (simulated)

## Simulation entities

- ### RequestGenerator:
    - This entity has the responsibility of generating loads from real  world logs and send them to LoadBalancer entity
  
- ### LoadBalancer:
    - Receive requests from LoadGenerator and balances them across cluster workers.
    - Every PERIOD, sends application metrics (response time) to Monitoring entity.
    - It also forks a QueueProcess having access to request queue.
  
- ### QueueProcess:
    - Has access to request queue of LoadBalancer. It's responsibility is to receive request responses returned by pods, 
  and send a request in the queue to pods if exists any.
  
- ### Monitoring:
    - Receives application metrics from LoadBalancer
    - Collects cluster metrics (cpu utilization) through kubernetes api

- ### Recommender:
    - Each RECOMMENDATION_PERIOD, queries Monitoring to receive application and cluster metrics.
    - Using values received from Monitoring, Recommends new replica count for the application through kubernetes api.

## Notes
 - ### [simpy](https://simpy.readthedocs.io/en/latest/) is used to simulate whole process.
 - ### kubernetes cluster is implemented as a [gym](https://github.com/openai/gym) environment.
---
###### Master Thesis Project - computer engineering department of IUST - 2021 summer
