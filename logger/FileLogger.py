import logging
import builtins
import os.path as path

class FileLogger():
    def __init__(self,
                 file_path,
                 file_name):
        self.origin_print = builtins.print

        target_path = path.join(file_path, file_name)
        
        # 로깅 설정
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # 콘솔 출력 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 파일 출력 핸들러 설정
        file_handler = logging.FileHandler(target_path)
        file_handler.setLevel(logging.INFO)

        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 핸들러 추가
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        self.logger = logger

        
    # 기존의 print를 오버라이드하여 logger.info로 출력하도록 설정
    def custom_print(self, *args, **kwargs):
        message = ' '.join(map(str, args))  # print의 인자를 문자열로 합침
        self.logger.info(message)  # 로그로 출력

    def on(self):
        builtins.print = self.custom_print
    
    def off(self):
        builtins.print = self.origin_print
