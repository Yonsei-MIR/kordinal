from aiohttp import ClientSession

from kordinal.client.client_utils import convert_data_from_client_to_request
from kordinal.logger_config import logger


class AsyncOpenAI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key

    async def create_api_request(self, session: ClientSession, endpoint: str, data: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        async with session.post(url, json=data, headers=headers) as resp:
            response = await resp.json()
            return response

    async def v1_chat_completion(self, session: ClientSession, data: dict) -> tuple[dict, str]:
        json_data = convert_data_from_client_to_request(data)
        # json_data["request_id"] = request_id
        try:
            response = await self.create_api_request(session, 'v1/chat/completions', json_data)
        except Exception as e:
            logger.error(f"Exception in v1_chat_completion: {e}", exc_info=True)
            raise
        return response

    
    async def v1_completion(self, session: ClientSession, data: dict) -> tuple[dict, str]:
        json_data = convert_data_from_client_to_request(data)
        try:
            response = await self.create_api_request(session, 'v1/completions', json_data)
        except Exception as e:
            logger.error(f"Exception in v1_completion: {e}", exc_info=True)
            raise
        return response

    async def chat_completion(self, session: ClientSession, data: dict) -> tuple[dict, str]:
        json_data = convert_data_from_client_to_request(data)
        try:
            response = await self.create_api_request(session, 'chat/completions', json_data)
        except Exception as e:
            logger.error(f"Exception in chat_completion: {e}", exc_info=True)
            raise
        return response
    
    async def completion(self, session: ClientSession, data: dict) -> tuple[dict, str]:
        json_data = convert_data_from_client_to_request(data)
        try:
            response = await self.create_api_request(session, 'completions', json_data)
        except Exception as e:
            logger.error(f"Exception in completion: {e}", exc_info=True)
            raise
        return response




