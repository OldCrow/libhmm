#ifndef LIBHMM_H_
#define LIBHMM_H_

#include "libhmm/common/common.h"
#include "libhmm/common/distribution_traits.h"
#include "libhmm/hmm.h"

#include "libhmm/distributions/distributions.h"

#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/scaled_forward_backward_calculator.h"
#include "libhmm/calculators/log_forward_backward_calculator.h"

#include "libhmm/training/viterbi_trainer.h"
#include "libhmm/training/segmented_kmeans_trainer.h"
#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/training/scaled_baum_welch_trainer.h"

#include "libhmm/io/file_io_manager.h"
#include "libhmm/io/xml_file_reader.h"
#include "libhmm/io/xml_file_writer.h"

// Performance optimization components
#include "libhmm/performance/simd_support.h"
#include "libhmm/performance/thread_pool.h"
#include "libhmm/calculators/optimized_forward_backward_calculator.h"
#include "libhmm/calculators/calculator_traits.h"

#include "libhmm/two_state_hmm.h"

#endif
