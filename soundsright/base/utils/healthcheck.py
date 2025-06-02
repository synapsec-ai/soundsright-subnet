"""
This module implements a health check API for the LLM Defender Subnet
neurons. The purpose of the health check API is to provide key
information about the health of the neuron to enable easier
troubleshooting. 

It is highly recommended to connect the health check API into the
monitoring tools used to monitor the server. The health metrics are not
persistent and will be lost if neuron is restarted.

Endpoints:
    /healthcheck/metrics
        Returns a dictionary of the metrics the health is derived from
    /healthcheck/events
        Returns list of relevant events related to the health metrics
        (error and warning)
        
Validator Endpoints:
    
    /healthcheck/current_models
        Returns information on models in current competition 
    /healthcheck/best_models
        Returns information on best models from last competition 
    /healthcheck/competitions 
        Returns information on which competitions are currently being 
        hosted by the subnet 
    /healthcheck/competition_scores
        Returns information about current competition-specific miner scores
    /healthcheck/scores
        Returns information about current overall miner scores
    /healthcheck/next_competition
        Returns timestamp of next competition time

Port and host can be controlled with --healthcheck_port and
--healthcheck_host parameters.
"""

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from typing import Dict
import datetime
import uvicorn
import threading
import numpy as np

class HealthCheckResponse(BaseModel):
    status: bool
    checks: Dict
    timestamp: str

class HealthCheckDataResponse(BaseModel):
    data: Dict | None | bool | list | str
    timestamp: str
    
class HealthCheckScoreResponse(BaseModel):
    data: np.ndarray | None | bool
    timestamp: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

class HealthCheckAPI:
    def __init__(self, host: str, port: int, is_validator: bool, current_models: dict | None = None, best_models: dict | None = None):

        # Variables
        self.host = host
        self.port = port
        self.is_validator = is_validator
        
        self.current_models = current_models
        self.best_models = best_models
        self.competition_scores = None
        self.scores = None
        self.next_competition_timestamp = None
        
        # Status variables
        self.health_metrics = {
            "start_time": datetime.datetime.now().timestamp(),
            "neuron_running": False,
            "iterations": 0,
            "datasets_generatred":0,
            "competitions_judged":0,
            "log_entries.success": 0,
            "log_entries.warning": 0,
            "log_entries.error": 0,
            "axons.total_filtered_axons": 0,
            "axons.total_queried_axons": 0,
            "axons.queries_per_second":0.0,
            "responses.total_valid_responses": 0,
            "responses.total_invalid_responses": 0,
            "responses.valid_responses_per_second":0.0,
            "responses.invalid_responses_per_second":0.0,
            "weights.targets": 0,
            "weights.last_set_timestamp": None,
            "weights.last_committed_timestamp":None,
            "weights.last_revealed_timestamp":None,
            "weights.total_count_set":0,
            "weights.total_count_committed":0,
            "weights.total_count_revealed":0,
            "weights.set_per_second":0.0,
            "weights.committed_per_second":0.0,
            "weights.revealed_per_second":0.0
        }

        self.health_events = {
            "warning": [],
            "error": [],
            "success": []
        }

        # App
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):
        self.app.add_api_route(
            "/healthcheck/metrics",
            self._healthcheck_metrics,
            response_model=HealthCheckDataResponse,
        )
        self.app.add_api_route(
            "/healthcheck/events",
            self._healthcheck_events,
            response_model=HealthCheckDataResponse,
        )
        self.app.add_api_route(
            "/healthcheck/current_models",
            self._healthcheck_current_models,
            response_model=HealthCheckDataResponse,
        )
        self.app.add_api_route(
            "/healthcheck/best_models",
            self._healthcheck_best_models,
            response_model=HealthCheckDataResponse,
        )
        self.app.add_api_route(
            "/healthcheck/competitions",
            self._healthcheck_competitions,
            response_model=HealthCheckDataResponse,
        )
        self.app.add_api_route(
            "/healthcheck/competition_scores",
            self._healthcheck_competition_scores,
            response_model=HealthCheckDataResponse,
        )
        self.app.add_api_route(
            "/healthcheck/scores",
            self._healthcheck_scores,
            response_model=HealthCheckDataResponse,
        )
        self.app.add_api_route(
            "/healthcheck/next_competition",
            self._healthcheck_next_competition,
            response_model=HealthCheckDataResponse,
        )

    def _healthcheck_metrics(self):
        try:
            # Return the metrics collected by the HealthCheckAPI
            return {
                "data": self.health_metrics,
                "timestamp": str(datetime.datetime.now()),
            }
        except Exception:
            return {"data": None, "timestamp": str(datetime.datetime.now())}

    def _healthcheck_events(self):
        try:
            # Return the events collected by the HealthCheckAPI
            return {
                "data": self.health_events,
                "timestamp": str(datetime.datetime.now()),
            }
        except Exception:
            return {"data": None, "timestamp": str(datetime.datetime.now())}
        
    def _healthcheck_current_models(self):
        try: 
            return {
                "data": self.current_models if self.current_models else None,
                "timestamp": str(datetime.datetime.now()),
            }
        except Exception:
            return {"data": None, "timestamp": str(datetime.datetime.now())}
        
    def _healthcheck_best_models(self):
        try: 
            return {
                "data": self.best_models if self.best_models else None,
                "timestamp": str(datetime.datetime.now()),
            }
        except Exception:
            return {"data": None, "timestamp": str(datetime.datetime.now())}
        
    def _healthcheck_competitions(self):
        try: 
            competitions = [k for k in self.best_models]
            return {
                "data": competitions if competitions else None,
                "timestamp": str(datetime.datetime.now()),
            }
        except Exception:
            return {"data": None, "timestamp": str(datetime.datetime.now())}
        
    def _healthcheck_competition_scores(self):
        try:
            return {
                "data":self.competition_scores,
                "timestamp": str(datetime.datetime.now()),
            }
        except Exception:
            return {"data": None, "timestamp": str(datetime.datetime.now())}
        
    def _healthcheck_scores(self):
        try:
            return {
                "data":self.scores.tolist(),
                "timestamp": str(datetime.datetime.now()),
            }
        except Exception:
            return {"data": None, "timestamp": str(datetime.datetime.now())}
        
    def _healthcheck_next_competition(self):
        try:
            dt = datetime.datetime.fromtimestamp(self.next_competition_timestamp, tz=datetime.timezone.utc)
            formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S GMT')
            return {
                "data":formatted_timestamp,
                "timestamp": str(datetime.datetime.now()),
            }
        except Exception:
            return {"data": None, "timestamp": str(datetime.datetime.now())}

    def run(self):
        """This method runs the HealthCheckAPI"""
        threading.Thread(
            target=uvicorn.run,
            args=(self.app,),
            kwargs={"host": self.host, "port": self.port},
            daemon=True,
        ).start()

    def add_event(self, event_name: str, event_data: str) -> bool:
        """This method adds an event to self.health_events dictionary"""
        if isinstance(event_name, str) and event_name.upper() in (
            "SUCCESS",
            "ERROR",
            "WARNING",
        ):

            # Append the received event under the correct key if it is str
            if isinstance(event_data, str) and not isinstance(event_data, bool):
                event_severity = event_name.lower()
                self.health_events[event_severity].append(
                    {"timestamp": str(datetime.datetime.now()), "message": event_data}
                )

                # Reduce the number of events if more than 250
                if len(self.health_events[event_severity]) > 250:
                    self.health_events[event_severity] = self.health_events[
                        event_severity
                    ][-250:]

                return True

        return True

    def append_metric(self, metric_name: str, value: int | bool) -> bool:
        """This method increases the metric counter by the value defined
        in the counter. If the counter is bool, sets the metric value to
        the provided value. This function must be executed whenever the
        counters for the given metrics wants to be updated"""

        if metric_name in self.health_metrics.keys() and value > 0:
            if isinstance(value, bool):
                self.health_metrics[metric_name] = value
            else:
                self.health_metrics[metric_name] += value
        else:
            return False

        return True
    
    def update_metric(self, metric_name: str, value: str | int | float):
        """This method updates a value for a metric that renews every iteration."""
        if metric_name in self.health_metrics.keys():
            self.health_metrics[metric_name] = value 
            return True 
        else: 
            return False

    def update_rates(self):
        """This method updates the rate-based parameters within the 
        healthcheck API--prompts generated per second, axons queried per
        second, valid responses per second and invalid responses per second."""

        time_passed = datetime.datetime.now().timestamp() - self.health_metrics['start_time']

        if time_passed > 0:

            # Calculate queries per second 
            self.health_metrics['axons.queries_per_second'] = self.health_metrics['axons.total_queried_axons'] / time_passed

            # Calculate valid responses per second 
            self.health_metrics['responses.valid_responses_per_second'] = self.health_metrics['responses.total_valid_responses'] / time_passed

            # Calculate invalid responses per second 
            self.health_metrics['responses.invalid_responses_per_second'] = self.health_metrics['responses.total_invalid_responses'] / time_passed

            # Calculate weight set events per second 
            self.health_metrics['weights.set_per_second'] = self.health_metrics['weights.total_count_set'] / time_passed

            # Calculate weight commit events per second
            self.health_metrics['weights.committed_per_second'] = self.health_metrics["weights.total_count_committed"] / time_passed

            # Calculate weight reveal events per second 
            self.health_metrics['weights.revealed_per_second'] = self.health_metrics["weights.total_count_revealed"] / time_passed
            
            return True
        
        else:
            return False
        
    def update_current_models(self, current_models):
        self.current_models = current_models 
        
    def update_best_models(self, best_models): 
        self.best_models = best_models
    
    def update_competition_scores(self, competition_scores):
        self.competition_scores = competition_scores
        
    def update_scores(self, scores):
        self.scores = scores

    def update_next_competition_timestamp(self, next_competition_timestamp):
        self.next_competition_timestamp = next_competition_timestamp