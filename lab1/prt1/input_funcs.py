from math import prod, fsum

f_types = ["min",
			"max",
			"sum",
			"prod"]


def get_input_fun(f_type):
	if not isinstance(f_type, str):
		raise Exception("f_type must be str object from list above:\n"+str(f_types))
	f_type = f_type.lower()
	if f_type not in f_types:
		raise Exception("f_type must be one from list above:\n"+str(f_types))

	if f_type == "min":
		return min
	elif f_type == "max":
		return max
	elif f_type == "sum":
		return fsum
	elif f_type == "prod":
		return prod
