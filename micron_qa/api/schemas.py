from pydantic import BaseModel
from typing import List

class TabularInput(BaseModel):
    features: List[float]

