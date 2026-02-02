### ADI/O: An AI-Assisted Therapeutic Platform for Autism-Focused Reading Comprehension 

## Overview

ADI/O is an AI-assisted therapeutic tool that operationalizes dual-coding based therapy to improve reading comprehension for individuals with Autism Spectrum Disorder (ASD). The system combines Automatic Speech Recognition (ASR) with multimodal visual context to provide accurate transcription of disordered speech in therapeutic settings. By integrating visual cues with speech recognition, ADI/O enables scalable, low-cost digital therapies for individuals who currently lack access to in-person treatment.

## System Architecture

ADI/O follows a modular architecture with five core components that work together to deliver personalized, context-aware speech therapy. The system is designed to replicate the authentic interactive experience central to dual-coding based therapy, where children are shown visual prompts and guided toward describing them in detail. Each component plays a specific role in creating a comprehensive therapeutic environment that adapts to individual needs while maintaining the structured approach of evidence-based interventions.

## Core Components

### 1. Image Bank

The Image Bank serves as the foundation of the visual-verbal integration exercises. It contains a curated collection of 100 high-quality, visually clear images that are rich in descriptive features yet simple enough to avoid overwhelming users. Each image is carefully selected to support the therapeutic goals of developing comprehension through guided verbalization.

Images are organized by scene type, including indoor, outdoor, and social settings. Each image is assigned a complexity score ranging from 1 to 2, allowing the system to progressively challenge users as they develop their descriptive skills. The images are also tagged with entities present in the scene, such as people, animals, objects, and environmental elements. This metadata enables the system to generate appropriate questions and evaluate responses based on the specific content of each image.

The collection includes diverse scenarios that encourage rich descriptions—from a child carrying an umbrella in the rain to siblings exploring a garden. These images are designed to elicit responses that can be evaluated across multiple dimensions, supporting the development of both verbal expression and visual comprehension skills.

### 2. Gold-Standard Description Generator

The Gold-Standard Description Generator uses a large language model to create detailed, developmentally appropriate image descriptions that serve as reference standards for evaluating user responses. These descriptions are organized according to the structure word framework used in Visualization and Verbalization (V/V) therapy, which emphasizes key elements such as what, where, who, color, size, and mood.

For each image, the generator produces a comprehensive set of structured comprehension questions aligned with the V/V framework. For instance, if an image depicts a child carrying an umbrella in the rain, the system might generate questions such as "Who is in the image?", "What is the person doing?", and "What color is the umbrella?" Each question is accompanied by an ideal descriptive answer that represents the gold standard for evaluation.

The generator ensures that all descriptions and questions are developmentally appropriate, using language and concepts suitable for the target age group. This creates a consistent baseline against which user responses can be measured, enabling accurate assessment of progress and identification of areas needing additional support.

### 3. ASR Transcription Module

The ASR Transcription Module is the core technological innovation of ADI/O, designed to achieve near-perfect accuracy in transcribing disordered speech. This module addresses the critical challenge that existing ASR systems face when processing atypical speech patterns, including slurring, atypical articulation, and inconsistent rhythm.

The module combines three key technologies to achieve superior performance. First, it uses Whisper Small, OpenAI's advanced ASR system that has been trained on large-scale multilingual datasets, making it robust across different accents and dialects. Second, the system employs Low-Rank Adaptation (LoRA) to fine-tune the Whisper model specifically for disordered speech. This parameter-efficient approach allows the model to adapt to domain-specific speech patterns while maintaining computational efficiency.

The fine-tuning process uses Google's Project Euphonia dataset, which contains over one million utterances and more than 1,300 hours of speech from speakers with diverse speech disorders. This extensive dataset enables the model to generalize across different disorder types, accents, and speaking conditions, capturing the variations that characterize atypical speech.

The third and most innovative component is the integration of CLIP (Contrastive Language-Image Pre-training), which enables multimodal decoder biasing. CLIP is a model that learns to map images and text into a shared embedding space, capturing their semantic relationships. This capability allows the ASR system to use visual context to improve transcription accuracy.

The multimodal process works as follows: when a user describes an image, the audio input is first processed by the fine-tuned Whisper model, which generates multiple candidate transcriptions. Simultaneously, the image is converted into a visual embedding using CLIP's vision encoder. Each ASR candidate transcription is also converted into a text embedding using CLIP's text encoder. The system then computes the cosine similarity between the image embedding and each text embedding, measuring how closely the semantic meaning of each candidate transcription matches the visual context.

These similarity scores are combined with the original ASR confidence scores through a fusion coefficient that balances speech model confidence with visual context. The candidates are then rescored, with the highest-scoring option selected as the final transcription. This approach is particularly effective for resolving ambiguities in descriptive speech—for example, distinguishing between phonetically similar words like "bear" and "bare" when the visual context clearly shows an animal.

Audio preprocessing is an essential component of the ASR pipeline. Raw audio undergoes cleaning to remove background noise and microphone artifacts, which disproportionately affect atypical speech. Audio formats are normalized to ensure consistency across different speakers and recording conditions. The system also employs data augmentation techniques including controlled noise injection, pitch shifting, and time-stretching to improve overall model robustness and performance.

### 4. Therapeutic Scaffolding Module

The Therapeutic Scaffolding Module generates progressive, scaffolded questions that guide users toward detailed and comprehensive descriptions. This module uses a large language model to create questions based on the structure word framework, breaking down complex scenes into manageable components that children can understand and articulate more effectively.

The module adapts its questioning strategy based on user responses, providing increasingly detailed prompts as users demonstrate their ability to describe visual content. It offers model example responses that help users understand the level of detail expected, supporting the development of cognitive skills through guided verbalization.

Questions are organized around key structure words including who, what, where, color, size, shape, and mood. This structured approach ensures that users learn to systematically observe and describe visual information, strengthening the connection between visual and verbal understanding that is central to dual-coding theory.

The scaffolding is designed to be supportive rather than prescriptive, allowing children to express themselves freely while providing gentle guidance toward more complete and coherent descriptions. This balance between structure and flexibility is essential for maintaining engagement while promoting skill development.

### 5. User Performance Evaluation Module

The User Performance Evaluation Module assesses user responses against the gold-standard descriptions generated for each image. This evaluation occurs across four main criteria that capture different aspects of descriptive quality and accuracy.

The first criterion is Omission, which measures whether users are missing key details that should be present in their descriptions based on the gold standard. The second is Extra Details, which tracks additional information that users provide beyond what is in the reference description. This metric helps identify when users are making creative observations or potentially misinterpreting visual content.

The third criterion is Detail and Clarity, which evaluates the depth and precision of descriptions. This measures not just what users describe, but how clearly and specifically they articulate their observations. The fourth criterion is Linguistic Accuracy, which assesses grammar, syntax, and word choice, ensuring that users are developing both comprehension skills and language proficiency.

The module generates quantitative scores for each criterion, providing both immediate feedback and long-term progress tracking. This data enables the system to identify patterns in user performance, adapt the difficulty of exercises, and provide targeted support where needed. The evaluation approach is designed to be encouraging and constructive, focusing on growth and improvement rather than simply identifying errors.

## System Workflow

A typical therapeutic session with ADI/O follows a structured yet flexible workflow designed to maximize learning outcomes. The session begins when a user initiates an interaction, at which point the system selects an appropriate image from the Image Bank based on the user's current skill level and progress history.

If a gold-standard description has not yet been generated for the selected image, the Gold-Standard Description Generator creates one, organizing it according to the structure word framework. The image is then displayed to the user, and the Therapeutic Scaffolding Module generates an initial question designed to elicit a descriptive response.

When the user provides a verbal response, the audio is captured and preprocessed to remove noise and normalize the signal. The ASR Transcription Module then processes this audio through its multimodal pipeline. The fine-tuned Whisper model generates multiple candidate transcriptions, while CLIP encodes both the image and the text candidates into a shared semantic space. The system computes similarity scores and rescores the candidates, selecting the final transcription that best aligns with both the audio signal and the visual context.

The User Performance Evaluation Module then compares this transcription against the gold-standard description, generating scores across the four evaluation criteria. Based on these results, the Therapeutic Scaffolding Module adapts its next question, either probing for more detail in areas where information was omitted or moving to new aspects of the image if the user has demonstrated sufficient comprehension of the current elements.

This iterative process continues throughout the session, with the system dynamically adjusting its approach based on user performance. The integration of visual context throughout the transcription process ensures that even when speech is unclear or ambiguous, the system can leverage the image content to produce accurate transcriptions that enable meaningful evaluation and feedback.

## Training and Evaluation Approach

The development of ADI/O follows a progressive evaluation strategy that isolates the contributions of different system components. This approach begins with establishing a baseline by testing a validation dataset against the unmodified Whisper model and calculating its word error rate (WER). This baseline represents the performance of state-of-the-art general-purpose ASR on disordered speech in the therapeutic context.

The next stage involves fine-tuning Whisper on the Project Euphonia dataset using LoRA, which adapts the model to disordered speech patterns while maintaining computational efficiency. The same validation set is then re-evaluated to measure the improvement attributable specifically to fine-tuning, providing a clear metric for the value of domain adaptation.

The final stage integrates CLIP-based image embeddings at inference time, allowing the ASR output to be evaluated jointly with visual context. This multimodal integration is expected to further reduce word error rates by resolving ambiguities that arise when speech is unclear but visual context provides strong semantic cues.

The validation dataset itself is carefully constructed to represent the type of speech encountered in therapeutic use. The curated image set is provided to a large language model, which generates structured comprehension questions and gold-standard answers for each image, grounded in the structure word framework. Target speakers with disordered speech then record these answers, producing an aligned dataset of image and audio pairs that serves as the benchmark for evaluating model performance.

This progressive validation approach ensures that each enhancement—fine-tuning and multimodal integration—can be evaluated quantitatively, providing clear evidence of the system's improvements over baseline performance. Word error rate serves as the primary metric, with the expectation that the combination of fine-tuning and multimodal context will achieve the near-perfect accuracy required for therapeutic applications.

## Key Design Decisions

Several important design decisions shape the architecture and implementation of ADI/O. The choice to use LoRA rather than full fine-tuning reflects a balance between performance and computational cost. LoRA enables parameter-efficient adaptation that captures domain-specific patterns while remaining practical for deployment in resource-constrained settings.

The integration of CLIP for multimodal alignment leverages pre-trained vision-language understanding rather than training a custom multimodal model from scratch. This approach capitalizes on CLIP's ability to map images and text into a shared semantic space, enabling effective visual-semantic alignment for decoder biasing.

The modular architecture of the system enables independent development and testing of components, allowing each module to be optimized separately before integration. This design also facilitates future enhancements and modifications, as components can be updated or replaced without requiring system-wide changes.

The adoption of the structure word framework aligns the system with proven V/V therapeutic methodology, ensuring that ADI/O remains grounded in evidence-based practices while leveraging cutting-edge AI capabilities. This alignment is crucial for maintaining therapeutic effectiveness while introducing technological innovation.

The progressive evaluation strategy isolates improvements from fine-tuning versus multimodal integration, providing clear evidence of each component's contribution. This approach supports both scientific rigor and practical understanding of which enhancements provide the greatest value.

## Development Phases

The development of ADI/O is organized into five phases, with the first phase currently in progress. Phase 1 focuses on assembling the imagery dataset and establishing the visual framework, including finalizing the structure-word mapping that links each image to different question types. This phase also involves validating the dataset for coverage, clarity, and alignment with verbal prompts.

Phase 2 will address audio dataset preparation and ASR modeling. This includes cleaning, normalizing, and labeling audio files with metadata such as speaker type and recording conditions. The phase will also involve collecting human samples of atypical speech, running them against the baseline Whisper model to establish performance metrics, and then fine-tuning the model using LoRA to measure improvements in transcription quality.

Phase 3 focuses on multimodality development, creating a system that integrates visual cues with transcribed speech. This phase involves generating CLIP embeddings for images, combining them with ASR transcript candidates, and testing different fusion coefficients to determine how visual context changes transcript ranking and improves accuracy.

Phase 4 will combine all elements into a working therapy application prototype, defining how the system presents images, asks questions, and delivers automated feedback. This phase includes testing for stability and basic interaction flow, resulting in a functional version ready for full evaluation.

Phase 5 involves final testing and reporting, running end-to-end tests to measure transcription accuracy, multimodal gains, and overall interaction quality. This phase will gather example transcripts, scoring comparisons, and key error cases to support results, presenting findings with clear documentation and visualizations.

## Future Enhancements

Several enhancements are planned for future development. Personalization capabilities would enable speaker-specific adaptation, allowing the system to learn individual speech patterns and further improve accuracy. Real-time processing optimizations would reduce latency, making interactions feel more natural and responsive.

Gamification features could increase engagement, particularly for younger users, by incorporating game-like elements that make therapy sessions more enjoyable while maintaining therapeutic effectiveness. Enhanced progress tracking would provide long-term analytics, helping users, families, and therapists understand development over extended periods.

Multi-language support would expand accessibility, making ADI/O available to diverse populations worldwide. This enhancement would require not only multilingual ASR capabilities but also culturally appropriate images and questions that resonate with different communities.

## Conclusion

ADI/O represents an innovative approach to making evidence-based reading comprehension therapy accessible through AI technology. By combining domain-specific fine-tuning with multimodal adaptation, the system addresses the critical challenge of accurately transcribing disordered speech in therapeutic contexts. The integration of visual context with speech recognition enables the system to resolve ambiguities and achieve the accuracy necessary for effective digital therapy delivery.

The modular architecture and progressive evaluation strategy ensure that the system can be developed, tested, and improved systematically, with clear metrics for each component's contribution. By grounding the system in proven therapeutic methodologies while leveraging cutting-edge AI capabilities, ADI/O bridges the gap between technological innovation and equitable access to speech therapy services.

This work has the potential to unlock scalable, low-cost digital therapies for individuals who currently lack access to in-person treatment, making evidence-based interventions available to neurodiverse learners worldwide. The combination of accurate ASR, multimodal context understanding, and structured therapeutic scaffolding creates a comprehensive platform that can support reading comprehension development for individuals with Autism Spectrum Disorder and related conditions.

---

**Sidharth Bildikar**  
Freshman, Tesla STEM High School  
Redmond, Washington

*This project aims to bridge cutting-edge AI with equitable speech therapy delivery, making evidence-based interventions accessible to neurodiverse learners worldwide.*
