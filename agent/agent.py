from mm_user_proxy_agent import MultimodalUserProxyAgent
from autogen.agentchat import Agent
from typing import Dict, Optional, Union

class SketchpadUserAgent(MultimodalUserProxyAgent):
    
    def __init__(
        self,
        name,
        prompt_generator, 
        parser,
        executor,
        **config,
    ):
        super().__init__(name=name, **config)
        self.prompt_generator = prompt_generator
        self.parser = parser
        self.executor = executor
        
    def sender_hits_max_reply(self, sender: Agent):
        return self._consecutive_auto_reply_counter[sender.name] >= self._max_consecutive_auto_reply

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """Receive a message from the sender agent.
        Once a message is received, this function sends a reply to the sender or simply stop.
        The reply can be generated automatically or entered manually by a human.
        """
        
        print("COUNTER:", self._consecutive_auto_reply_counter[sender.name])
        self._process_received_message(message, sender, silent)
        
        # parsing the code component, if there is one
        parsed_results = self.parser.parse(message)
        parsed_content = parsed_results['content']
        parsed_status = parsed_results['status']
        parsed_error_message = parsed_results['message']
        parsed_error_code = parsed_results['error_code']
        
        # if TERMINATION message, then return
        if not parsed_status and self._is_termination_msg(message):
            return
        
        # if parsing fails
        if not parsed_status:
            
            # reset the consecutive_auto_reply_counter
            if self.sender_hits_max_reply(sender):
                self._consecutive_auto_reply_counter[sender.name] = 0
                return
            
            # if parsing fails, construct a feedback message from the error code and message of the parser
            # send the feedback message, and request a reply
            self._consecutive_auto_reply_counter[sender.name] += 1
            reply = self.prompt_generator.get_parsing_feedback(parsed_error_message, parsed_error_code)
            self.feedback_types.append("parsing")
            self.send(reply, sender, request_reply=True)
            return
        
        # if parsing succeeds, then execute the code component
        if self.executor:
            # go to execution stage if there is an executor module
            exit_code, output, file_paths = self.executor.execute(parsed_content)
            reply = self.prompt_generator.get_exec_feedback(exit_code, output)
            
            # if execution fails
            if exit_code != 0:
                if self.sender_hits_max_reply(sender):
                    # reset the consecutive_auto_reply_counter
                    self._consecutive_auto_reply_counter[sender.name] = 0
                    return
                
                self._consecutive_auto_reply_counter[sender.name] += 1
                self.send(reply, sender, request_reply=True)
                return
                
            # if execution succeeds
            else:
                self.send(reply, sender, request_reply=True)
                self._consecutive_auto_reply_counter[sender.name] = 0
                return
    
    def generate_init_message(self, query, n_image):
        content = self.prompt_generator.initial_prompt(query, n_image)
        return content

    def initiate_chat(self, assistant, message, n_image, task_id, log_prompt_only=False):
        self.current_task_id = task_id
        self.feedback_types = []
        initial_message = self.generate_init_message(message, n_image)
        if log_prompt_only:
            print(initial_message)
        else:
            assistant.receive(initial_message, self, request_reply=True)