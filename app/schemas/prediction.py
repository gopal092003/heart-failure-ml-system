from pydantic import BaseModel, Field


class PatientData(BaseModel):
    age: float = Field(..., ge=0, le=120)
    anaemia: int = Field(..., ge=0, le=1)
    creatinine_phosphokinase: float = Field(..., ge=0)
    diabetes: int = Field(..., ge=0, le=1)
    ejection_fraction: float = Field(..., ge=0, le=100)
    high_blood_pressure: int = Field(..., ge=0, le=1)
    platelets: float = Field(..., ge=0)
    serum_creatinine: float = Field(..., ge=0)
    serum_sodium: float = Field(..., ge=0, le=200)
    sex: int = Field(..., ge=0, le=1)
    smoking: int = Field(..., ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "age": 65,
                "anaemia": 0,
                "creatinine_phosphokinase": 582,
                "diabetes": 0,
                "ejection_fraction": 38,
                "high_blood_pressure": 1,
                "platelets": 263358.03,
                "serum_creatinine": 1.9,
                "serum_sodium": 130,
                "sex": 1,
                "smoking": 0
            }
        }