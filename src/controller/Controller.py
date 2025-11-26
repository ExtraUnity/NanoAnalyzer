from src.shared.Commands import Command
from src.shared.RequestError import RequestError
from src.controller.RequestHandler import RequestHandler
import threading
class Controller():
    def __init__(self, pre_loaded_model_name=None):
        self.request_handler = RequestHandler(pre_loaded_model_name)
        self.commands = {
            Command.SEGMENT: self.request_handler.process_request_segment,
            Command.RETRAIN: self.request_handler.process_request_train,
            Command.FINE_TUNE: self.request_handler.process_request_fine_tune,
            Command.LOAD_MODEL: self.request_handler.process_request_load_model,
            Command.TEST_MODEL: self.request_handler.process_request_test_model,
            Command.SEGMENT_FOLDER: self.request_handler.process_request_segment_folder,
            Command.LOAD_IMAGE: self.request_handler.process_request_load_image,
        }
    
    def process_command(self, command, *args, **kwargs):
        """Routes the command to the appropriate backend method."""
        if command not in self.commands:
            print(f"Unknown command: {command}")
            return
        
        try:
            data = self.commands[command](*args, **kwargs)
            return data
        except Exception as e:
            return RequestError(str(e))

            
            
