from pydantic import BaseModel

class GetYtLinks(BaseModel):
    links : list[str] 

    class Config:
        from_attributes = True