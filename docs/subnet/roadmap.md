# Subnet Roadmap

The current goal for the subnet is to facilitate the open-source research and development of state-of-the-art speech enhancement models. We recognize that there is potential to create far more open-source work in this field.

The ultimate goal of the subnet is to create a monetized product in the form of an API. However, in order to make the product as competetive as possible, the subnet's first goal is to create a large body of work for miners to draw their inspiration from.

The following roadmap outlines our plans to bring a SoTA speech enhancement API into fruition:

## Versioning and release management
In order to ensure the subnet users can prepare in advance we have defined a formal patching policy for the subnet components.

The subnet uses **semantic versioning** in which the version number consists of three parts (Major.Minor.Patch). Depending on the type of release, there are a few things that the subnet users should be aware of.

- Major Releases (**X**.0.0)
    - There can be breaking changes and updates are mandatory for all subnet users.
    - After the update is released, the `weights_version` hyperparameter is adjusted immediately after release such that in order to set the weights in the subnet, the neurons must be running the latest version.
    - Major releases are communicated in the Subnet's Discord channel at least 1 week in advance.
    - Registration may be disabled for up to 24 hours.

- Minor releases (0.**X**.0)
    - There can be breaking changes.
    - In case there are breaking changes, the update will be announced in the Subnet's Discord channel at least 48 hours in advance. Otherwise a minimum of 24 hour notice is given.
    - If there are breaking changes, the `weights_version` hyperparameter is adjusted immediately after release such that in order to set the weights in the subnet, the neurons must be running the latest version.
    - If there are no breaking changes, the `weights_version` hyperparameter will be adjusted 24 hours after the launch.
    - Minor releases are mandatory for all subnet users.
    - Registration may be disabled for up to 24 hours.

- Patch releases (0.0.**X**)
    - Patch releases do not contain breaking changes and updates will not be mandatory unless there is a need to hotfix either scoring or penalty algorithms.
    - Patch releases without changes to scoring or penalty algorithms are pushed to production without prior notice.

## SoundsRight v1.0.0
- Register on testnet
- 16 kHz competitions for denoising and dereverberation tasks

## SoundsRight v1.1.0
- Register on mainnet

## SoundsRight v2.0.0
- TTS generation upgrade
- 48 kHz competitions for denoising and dereverberation tasks

## SoundsRight v3.0.0 
- More utilities provided to miners and validators
- Validator performance dashboards

## SoundsRight v4.0.0 
- Complete subnet overhaul to focus on monetization via API 