from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    data_folder = Path('../../data/')
    df          = pd.read_csv(data_folder / 'raw' / 'FACTDATA_SEP2018.TXT')
    output_path = data_folder / 'interim' / 'fed_emp.csv'

    df.EDLVL[df.EDLVL == '**']      = float('nan')
    df.EDLVL                        = df.EDLVL.astype(float)
    df.GSEGRD[df.GSEGRD == '**']    = float('nan')
    df.GSEGRD                       = df.GSEGRD.astype(float)
    df.OCC[df.OCC == '****']        = float('nan')
    df.OCC                          = df.OCC.astype(float)
    df.SUPERVIS[df.SUPERVIS == '*'] = float('nan')
    df.SUPERVIS                     = df.SUPERVIS.astype(float)
    df.TOA[df.TOA == '**']          = float('nan')
    df.TOA                          = df.TOA.astype(float)
    df                              = df.drop(['DATECODE', 'EMPLOYMENT'], axis = 1)

    df.to_csv(output_path, index = False)
