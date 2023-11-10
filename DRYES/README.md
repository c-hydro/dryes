# DRYES

This is an internal Python package for setting up operational data analysis pipelines. The package is designed to handle raster maps of meteorological or satellite quantities. The package follows a pipeline approach with the following steps:

1. **Obtain Data**: Fetch data from several different data sources each with their own API or similar type of request mechanisms.

2. **Preprocess Data**: Quality control and time aggregation to ensure data from various sources is in a common format and easy to find.

3. **Index Calculation**: Calculate climatological indices. Some indices will require the calculations of parameters as well (e.g., long term means or standard deviations).

4. **Post Processing**: Final quality control, smoothing of raster outputs and produce maps in the expected format.

## Installation

To install this package, navigate to the root directory of the project and run:

```bash
pip install .
```

## Usage

Import the package in your Python script as follows:

```python
import DRYES
```

You can then use the functions and classes provided by the package to set up your data analysis pipeline.

## Testing

To run the tests, navigate to the root directory of the project and run:

```bash
python -m unittest discover tests
```

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.