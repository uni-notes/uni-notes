## Task $T$

Process of learning itself is not the task; learning is the means of attaining ability to perform the task

Usually described in terms of how the machine
learning system should process an instance (collection of features), which is usually represented as a vector.

|                                  |                                                              | Function Mapping           | Example                                                      |
| -------------------------------- | ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ |
| Regression                       | Predicting a continuous numerical output                     | $R^n \to R$                | Stock value prediction                                       |
| Classification                   | Categorizing input into a discrete output<br/>or outputing a probability dist over classes<br />Derived from regression | $R^n \to \{1, \dots, k \}$ | Categorizing images<br />Fraud detection                     |
| Classification w/ missing inputs | Learn distribution over all variables, solve by marginalizing over missing variables | $R^n \to \{1, \dots, k \}$ |                                                              |
| Clustering                       | Grouping inputs into clusters                                |                            | Grouping similar images                                      |
| Transcription                    | Convert unstructured data intro discrete textual form        |                            | OCR<br />Speech Recognition                                  |
| Machine Translation              | Convert it<br/>into a sequence of symbols into another language |                            | Natural Language Translation                                 |
| Structured Output                | Output data structure has<br/>relationships between elements |                            | Parsing<br />Image segmentation<br />Image captioning        |
| Anomaly Detection                | Identify abnormal events                                     |                            | Fraud detection                                              |
| Synthesis & Sampling             | Generate new samples similar to those in<br/>training data   |                            | Texture generation<br />Speech synthesis<br />Supersampling images |
| Data Imputation                  | Predict values of missing entries                            |                            |                                                              |
| Denoising                        | Predict clean output from corrupt input                      |                            | Image/Video denoising                                        |
| Density Estimation               | Identify underlying probability distribution of set of inputs |                            |                                                              |

