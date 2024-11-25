#include <cstdarg>
#include <cstdint>

namespace Eloquent {
    namespace ML {
        namespace Port {
            class LogisticRegression {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        float votes[3] = { 28.788070118403 ,6.127261044079 ,-34.915331162435  };
                        votes[0] += dot(x,   0.755488591112 );
                        votes[1] += dot(x,   0.011512450137 );
                        votes[2] += dot(x,   -0.76700104129 );
                        // return argmax of votes
                        uint8_t classIdx = 0;
                        float maxVotes = votes[0];

                        for (uint8_t i = 1; i < 3; i++) {
                            if (votes[i] > maxVotes) {
                                classIdx = i;
                                maxVotes = votes[i];
                            }
                        }

                        return classIdx;
                    }

                    /**
                    * Predict readable class name
                    */
                    const char* predictLabel(float *x) {
                        return idxToLabel(predict(x));
                    }

                    /**
                    * Convert class idx to readable name
                    */
                    const char* idxToLabel(uint8_t classIdx) {
                        switch (classIdx) {
                            case 0: return "Inturai_3";
                            case 1: return "Inturai_2";
                            case 2: return "Inturai_1";
                            default: return "Houston we have a problem";
                        }
                    }

                protected:
                    /**
                    * Compute dot product
                    */
                    float dot(float *x, ...) {
                        va_list w;
                        va_start(w, x);
                        float dot = 0.0;

                        for (uint16_t i = 0; i < 1; i++) {
                            const float wi = va_arg(w, double);
                            dot += x[i] * wi;
                        }
                        va_end(w); // Clean up variadic arguments
                        return dot;
                    }
                };
            }
        }
    }
    int main() {
    Eloquent::ML::Port::LogisticRegression model;
    float features[] = { 1.0f }; // Example feature vector
    int prediction = model.predict(features);
    const char* label = model.predictLabel(features);

    return 0;
    }