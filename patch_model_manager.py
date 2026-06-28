path = r"D:\desk\X-AnyLabeling\anylabeling\services\auto_labeling\model_manager.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

new_block = """        elif model_config["type"] == "track_centerline":
            from .track_centerline import TrackCenterline

            try:
                model_config["model"] = TrackCenterline(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                template = "Error in loading model: {error_message}"
                translated_template = self.tr(template)
                error_text = translated_template.format(error_message=str(e))
                self.new_model_status.emit(error_text)
                logger.error(
                    f"Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
"""

old = """        else:
            raise Exception(f"Unknown model type: {model_config['type']}")"""

new = new_block + "\n" + old

if new in content:
    print("Block already exists")
elif old in content:
    content = content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Block inserted successfully")
else:
    print("ERROR: Could not find insertion point")
