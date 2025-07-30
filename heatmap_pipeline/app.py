from heatmap_pipeline import pred
import sys
from utils.helper import create_error, report
import os


def run(arguments):
    
    if not os.path.exists(str(arguments['input_annotation_path'])): 
        error = create_error(101, "coco_annotation_path does not exist.", arguments['coco_annotation_path'], __file__, sys._getframe().f_lineno)
        return report(success=False, error=error, summary_code=700)
    
    if not os.path.exists(str(arguments['image_path'])): 
        error = create_error(101, "image_path does not exist.", arguments['image_path'], __file__, sys._getframe().f_lineno)
        return report(success=False, error=error, summary_code=700)
    
    if not os.path.exists(str(arguments['output_dir'])):
        os.makedirs(arguments['output_dir']) 
    
    try:
        arguments['area_ratio'] = float(arguments['area_ratio']) or 0.2
    except:
        error = create_error(104, "area_ratio should be a float number.", arguments['area_ratio'], __file__, sys._getframe().f_lineno)
        return report(success=False, error=error, summary_code=700)

    try:
        arguments['sigma_ratio'] = float(arguments['sigma_ratio']) or 0.3
    except:
        error = create_error(104, "sigma_ratio should be a float number.", arguments['sigma_ratio'], __file__, sys._getframe().f_lineno)
        return report(success=False, error=error, summary_code=700)

    return pred.run(arguments)

def handler(event, context):
    try:
        return run(event)
    except Exception as e:
        exe_type, _, exc_tb = sys.exc_info()
        error = create_error(401, "An error occurred in handler function.", str(e), __file__, exc_tb.tb_lineno, exe_type)
        return report(success=False, error=error , summary_code=700)