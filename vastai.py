import time

from pydantic import BaseModel
from vastai_sdk import VastAI

from ssh import SSHConfig


class VastAIConfig(BaseModel):
    api_key: str
    gpu: str = "RTX_5090"


class VastAIInstance:
    def __init__(self, client: "VastAIClient", id: int):
        self.client = client
        self.id = id

    def get_ssh_config(self) -> SSHConfig:
        while True:
            instance_info = self.client.client.show_instance(id=self.id)
            ssh_host = str(instance_info["ssh_host"])
            ssh_port = int(instance_info["ssh_port"])
            if instance_info["template_id"] is not None:
                ssh_port += 1
            if ssh_host and ssh_port:
                break
            time.sleep(1)
        return SSHConfig(host=ssh_host, port=ssh_port, user="root")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.client.destroy_instance(id=self.id)


class VastAIClient:
    def __init__(self, config: VastAIConfig):
        self.config = config
        self.client = VastAI(api_key=config.api_key)

    def get_offer(self) -> int:
        # Example method to get available instances based on GPU type
        offers = self.client.search_offers(
            type="on-demand",
            query=f"gpu_name={self.config.gpu} inet_up>=1024 inet_down>=1024 direct_port_count>=1 external=true rentable=true verified=true",
            order="dph",
            storage=50,
        )
        return offers[0]["id"]

    def create_instance(self, offer_id: int) -> VastAIInstance:
        # Example method to create an instance
        instance = self.client.create_instance(
            id=offer_id, disk=50, template_hash="305ac3ffd3e42e0d9ad1f4ae14729ec2"
        )

        instance_id = instance["new_contract"]
        self.client.start_instance(id=instance_id)
        return VastAIInstance(client=self, id=instance_id)
