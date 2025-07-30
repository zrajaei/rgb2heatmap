def run(arguments):
    try:
        from heatmap_pipeline.heatmap_generation import Heatmap
        from utils.helper import (create_error,
                            report,
                            read_json)
        import numpy as np
        import cv2
        import os
        import sys
            
        coco_annotation_path = arguments['input_annotation_path']
        input_image_path = arguments['image_path']
        output_image_path = os.path.join(arguments['output_dir'],os.path.basename(input_image_path))

        # Initialize the parameters
        area_ratio = arguments['area_ratio']
        sigma_ratio = arguments['sigma_ratio']


        coco_data = read_json(coco_annotation_path)
        image = cv2.imread(input_image_path)

        bboxes = []
        for annotation in coco_data['annotations']:
            bboxes.append(annotation['bbox'])
        bboxes = np.array(bboxes)

        heatmap = Heatmap()

        overlayed_heatmap = heatmap.run(image,
                                        bboxes,
                                        area_ratio=area_ratio,
                                        sigma_ratio=sigma_ratio)
        
        cv2.imwrite(output_image_path,overlayed_heatmap)

        return report(success=True, result=f'Saved heatmap overlay to: {output_image_path}')

    except Exception as e :
        exc_type, _, exc_tb = sys.exc_info()
        error = create_error(401, "An error occurred in run function.", str(e), __file__, exc_tb.tb_lineno, exc_type)
        return report(success=False, error=error, summary_code=700)