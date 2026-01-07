# python -m extract_tml.cli "inputs/UTM piping 20223sht33.jpg" --specs-allowed 7,9 --debug

# python -m extract_tml.cli "inputs/UTM piping 20223sht328.jpg" --specs-allowed 6 --debug

# python -m extract_tml.cli "inputs/UTM piping 20223sht394.jpg" --specs-allowed 5 --debug

# python -m extract_tml.cli "inputs/UTM piping 20223sht488.jpg" --specs-allowed 6 --debug

uvicorn extract_tml.app:app --host 0.0.0.0 --port 8000 --reload
