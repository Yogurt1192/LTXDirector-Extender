class FrameCountPassthrough:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT",),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "forward"
    CATEGORY = "LTXDirector/memory"

    def forward(self, value):
        return (int(value),)