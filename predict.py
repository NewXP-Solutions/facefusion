import subprocess
from cog import BasePredictor, Input, Path
import os

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load any necessary files or set up the environment if needed."""
        # self.model = load_model()

    def predict(
        self,
        source_image: Path = Input(description="Path to the source image (e.g., -s flag)"),
        target_video: Path = Input(description="Path to the target video (e.g., -t flag)"),
        output_path: Path = Input(description="Path to the output file (e.g., -o flag)"),
        execution_providers: str = Input(default="cuda", description="Execution providers (e.g., --execution-providers)"),
        processors: str = Input(default="face_swapper face_enhancer expression_restorer", description="Processors (e.g., --processors)"),
        face_swapper_model: str = Input(default="inswapper_128_fp16", description="Face swapper model (e.g., --face-swapper-model)"),
        face_enhancer_model: str = Input(default="gfpgan_1.4", description="Face enhancer model (e.g., --face-enhancer-model)"),
        expression_restorer_model: str = Input(default="live_portrait", description="Expression restorer model (e.g., --expression-restorer-model)"),
        frame_enhancer_model: str = Input(default=None, description="Frame enhancer model (e.g., --frame-enhancer-model)", required=False),
        frame_colorizer_model: str = Input(default=None, description="Frame colorizer model (e.g., --frame-colorizer-model)", required=False),
        face_editor_model: str = Input(default=None, description="Face editor model (e.g., --face-editor-model)", required=False),
        face_debugger_items: str = Input(default=None, description="Face debugger items (e.g., --face-debugger-items)", required=False)
    ) -> Path:
        """Run the face fusion model with the provided inputs."""

        # Ensure the output directory exists
        output_directory = os.path.dirname(output_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Created output directory at {output_directory}")

        # Construct the base command with mandatory arguments
        command = [
            "python", "facefusion.py", "headless-run",
            "-s", str(source_image),
            "-t", str(target_video),
            "-o", str(output_path),
            "--execution-providers", execution_providers,
            "--processors", processors,
            "--face-swapper-model", face_swapper_model,
            "--face-enhancer-model", face_enhancer_model,
            "--expression-restorer-model", expression_restorer_model
        ]

        # Dictionary of optional arguments
        optional_args = {
            "--frame-enhancer-model": frame_enhancer_model,
            "--frame-colorizer-model": frame_colorizer_model,
            "--face-editor-model": face_editor_model,
            "--face-debugger-items": face_debugger_items
        }

        # Add optional arguments only if they are provided (non-None)
        for flag, value in optional_args.items():
            if value:  # Add the flag and value only if the value is not None
                command.extend([flag, value])

        # Print the command for debugging purposes (optional)
        print("Running command: ", " ".join(command))

        # Run the subprocess command
        subprocess.run(command)

        # Return the output path where the result is stored
        return Path(output_path)
