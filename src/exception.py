import sys
import logger
import logging

def error_message_details(error,error_detail:sys):
    exception_type,exception,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    line_number=exc_tb.tb_lineno
    '''returns the tuple (type(e), e, e.__traceback__). That is, a tuple containing the type of the exception (a subclass of BaseException), 
    the exception itself, and a traceback object which typically encapsulates the call stack at the point where the exception last occurred.'''
    
    error_message="Erro occured in python script [{0}] line numer [{1}] error message [{2}]".format( file_name,line_number,str(error) )
    
    return error_message


class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
    

#code block to test if exception handling is working
"""    
if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("custom exception raised")
        raise CustomException(e,sys)
        
"""