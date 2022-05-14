from src.vessel_control.control.p_control import PControl
from src.vessel_control.shape.circle import Circle

pool_dim = (25, 10)

circle = Circle(number_of_checkpoints=20, radius=3, center=(pool_dim[0]/2, pool_dim[1]/2))
# vessel_control = PDControl(circle, position_threshold=1, orientation_threshold=1)
control = PControl(circle, position_threshold=1, orientation_threshold=1)

control.start()
