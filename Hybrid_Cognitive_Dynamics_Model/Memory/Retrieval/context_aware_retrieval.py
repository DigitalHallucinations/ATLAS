# modules/Hybrid_Cognitive_Dynamics_Model/Memory/context_aware_retrieval.py 

import numpy as np
from modules.Config.config import ConfigManager
from sklearn.metrics.pairwise import cosine_similarity
from modules.Hybrid_Cognitive_Dynamics_Model.SSM.state_space_model import StateSpaceModel

class ContextAwareRetrieval:
    def __init__(self, state_model: StateSpaceModel, config_manager: ConfigManager):
        self.state_model = state_model
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('ContextAwareRetrieval')

    async def get_context_vector(self):
        return await self.state_model.get_current_state_context()

    async def context_similarity(self, memory_context, current_context):
        similarity = cosine_similarity(memory_context.reshape(1, -1), current_context.reshape(1, -1))[0][0]
        return similarity

