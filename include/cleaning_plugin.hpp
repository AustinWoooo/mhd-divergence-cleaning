#pragma once

#include <string>
#include <vector>

#include "cleaning_plugin_utils.hpp"
#include "state.hpp"

struct CleaningStageContext {
    int nx = 0;
    int ny = 0;

    double dx = 0.0;
    double dy = 0.0;

    double time = 0.0;
    double dt_stage = 0.0;
    double gamma = 5.0 / 3.0;
};

class CleaningPlugin {
public:
    virtual ~CleaningPlugin() = default;

    virtual std::string name() const = 0;

    virtual bool usesPsi() const { return false; }
    virtual bool modifiesFlux() const { return false; }
    virtual bool hasSourceTerms() const { return false; }
    virtual bool hasCorrectionStep() const { return false; }

    virtual void beforeStage(
        std::vector<State>& U,
        const CleaningStageContext& ctx
    );

    virtual void modifyInterfaceFlux(
        State& flux,
        const State& UL,
        const State& UR,
        int direction,
        const CleaningStageContext& ctx
    );

    virtual void applySourceTerms(
        std::vector<State>& U,
        const CleaningStageContext& ctx
    );

    virtual void afterStage(
        std::vector<State>& U,
        const CleaningStageContext& ctx
    );

    virtual void afterStep(
        std::vector<State>& U,
        const CleaningStageContext& ctx
    );
};

class HyperbolicGlmPlugin final : public CleaningPlugin {
public:
    explicit HyperbolicGlmPlugin(double ch);

    std::string name() const override;
    bool usesPsi() const override;
    bool modifiesFlux() const override;

    void modifyInterfaceFlux(
        State& flux,
        const State& UL,
        const State& UR,
        int direction,
        const CleaningStageContext& ctx
    ) override;

    double ch() const { return ch_; }

private:
    double ch_;
};

class MixedGlmPlugin final : public CleaningPlugin {
public:
    MixedGlmPlugin(double ch, double cp);

    std::string name() const override;
    bool usesPsi() const override;
    bool modifiesFlux() const override;
    bool hasSourceTerms() const override;

    void modifyInterfaceFlux(
        State& flux,
        const State& UL,
        const State& UR,
        int direction,
        const CleaningStageContext& ctx
    ) override;

    void applySourceTerms(
        std::vector<State>& U,
        const CleaningStageContext& ctx
    ) override;

    double ch() const { return ch_; }
    double cp() const { return cp_; }

private:
    double ch_;
    double cp_;
};
