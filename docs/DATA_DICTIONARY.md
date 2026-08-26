# Data dictionary

| Column | Type / coding | Description |
| --- | --- | --- |
| `Gender (0: female, 1: male)` | Binary | Sex code used by the converted analysis file |
| `age(years)` | Integer | Age in years |
| `BMI` | Float, kg/m2 | Body mass index |
| `ap_hi(Systolic blood pressure)` | Integer, mmHg | Systolic blood pressure |
| `ap_lo(Diastolic blood pressure)` | Integer, mmHg | Diastolic blood pressure |
| `cholesterol(1: normal, 2: above normal, 3: well above normal)` | Ordinal 1-3 | Cholesterol category |
| `Glucose(1: normal, 2: above normal, 3: well above normal)` | Ordinal 1-3 | Glucose category |
| `smoking` | Binary | Smoking indicator |
| `alcohol` | Binary | Alcohol-use indicator |
| `active` | Binary | Physical-activity indicator |
| `cardiovascular disease` | Text label | `has cardiovascular disease` or `does not have cardiovascular disease` |

## Engineered features

| Feature | Formula |
| --- | --- |
| Pulse pressure (`PP`) | `SBP - DBP` |
| Mean arterial pressure (`MAP`) | `(SBP + 2 * DBP) / 3` |
| Risk count | Smoking + alcohol + inactivity + above-normal cholesterol + above-normal glucose |
