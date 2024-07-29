import os
import json
from pathlib import Path
from argparse import ArgumentParser

def main_evaluation_test(mainDir, rendersDir, gtProjDir, testMaskDir, mask):
    if mask:
        os.system(f"python ./myMetrics.py \
                    -m {str(mainDir)} \
                    -r {str(rendersDir)} \
                    -gt {str(gtProjDir)} \
                    --with_mask -p {str(testMaskDir)}")
    else:
        os.system(f"python ./myMetrics.py \
                    -m {str(newProjDir)} \
                    -gt {str(gtProjDir)}")
    
if __name__ == "__main__":
    #[cyw]:add path
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    test_dir = Path(args.model_paths[0]) / "test"

    # projDir = Path(".")/f"ours_30000" #! need too change
    # mask = True #! need too change
    for method in os.listdir(test_dir):
        projDir = test_dir / method
        rendersDir = projDir/"renders"
        gtProjDir = projDir/"gt"
        maskDir = projDir/"mask"
        mask = os.path.isdir(maskDir)
        print(mask )
        main_evaluation_test(projDir, rendersDir, gtProjDir, maskDir, mask)
        