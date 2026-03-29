import google.genai as genai
import time

from .Model import Model


class PaLM2(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        self.api_keys = api_keys
        api_pos = int(config["api_key_info"]["api_key_use"])
        if api_pos == -1: # use all keys
            self.key_id = 0
            self.api_key = None
        else: # only use one key at the same time
            assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
            self.api_key = api_keys[api_pos]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = None
        self.set_API_key()
        
    def set_API_key(self):
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client(api_key=self.api_keys[0])
        
    def query(self, msg):
        if not self.api_key:
            self.client = genai.Client(api_key=self.api_keys[self.key_id % len(self.api_keys)])
            self.key_id += 1
            
        try:
            response = self.client.models.generate_content(
                model=self.name,
                contents=msg,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }
            )
            result = response.text

        except Exception as e:
            print(e)
            if 'not supported' in str(e):
                return ''
            else:
                print('Error occurs! Please check output carefully.')
                time.sleep(300)
                return self.query(msg)
        
        if result == '' or result == None:
            result = 'Input may contain harmful content and was blocked by PaLM.'

        return result