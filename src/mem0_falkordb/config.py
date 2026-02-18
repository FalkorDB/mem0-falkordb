from typing import Optional

from pydantic import BaseModel, Field, model_validator


class FalkorDBConfig(BaseModel):
    host: str = Field("localhost", description="Host address for the FalkorDB server")
    port: int = Field(6379, description="Port for the FalkorDB server")
    database: str = Field("mem0", description="Graph name in FalkorDB")
    username: Optional[str] = Field(None, description="Username for FalkorDB authentication")
    password: Optional[str] = Field(None, description="Password for FalkorDB authentication")
    base_label: Optional[bool] = Field(True, description="Whether to use base node label __Entity__ for all entities")

    @model_validator(mode="before")
    def check_host_and_port(cls, values):
        if isinstance(values, dict):
            port = values.get("port")
            if port is not None and not isinstance(port, int):
                raise ValueError("'port' must be an integer.")
        return values
