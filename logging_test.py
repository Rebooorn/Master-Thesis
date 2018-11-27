import logging
import time

'''
this is the test for logging things
'''

logging.basicConfig(filename='example.log', format='%(asctime)s %(message)s', level=logging.INFO)

logging.info('new logging test start')
for i in range(20):
    logging.info('this is the {times} times of info logging'.format(times=i))
    logging.info('haha \r\t haha')