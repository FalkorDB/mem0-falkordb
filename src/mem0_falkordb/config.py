from typing import Optional

from pydantic import BaseModel, Field, model_validator


class FalkorDBConfig(BaseModel):
    host: str = Field("localhost", description="Host address for the FalkorDB server")
    port: int = Field(6379, description="Port for the FalkorDB server")
    database: str = Field(
        "mem0",
        description="Graph name prefix in FalkorDB (each user gets {database}_{user_id})",
    )
    username: Optional[str] = Field(
        None, description="Username for FalkorDB authentication"
    )
    password: Optional[str] = Field(
        None, description="Password for FalkorDB authentication"
    )
    base_label: Optional[bool] = Field(
        True, description="Whether to use base node label __Entity__ for all entities"
    )

    @model_validator(mode="after")
    def check_auth_pair(self):
        if bool(self.username) != bool(self.password):
            raise ValueError(
                "Both 'username' and 'password' must be provided together, or neither."
            )
        return self
