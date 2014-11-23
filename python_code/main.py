import sys
import logging
reload(logging)
logging.basicConfig(format = u'[%(asctime)s]  %(message)s', level = logging.INFO)
from RandomForestAdvanced import RandomForestAdvanced

if __name__ == "__main__":
	logging.info("Work Started")
	rfa = RandomForestAdvanced(sys.argv[1:])
	rfa.fit_and_test()
	logging.info("Work Finished")

