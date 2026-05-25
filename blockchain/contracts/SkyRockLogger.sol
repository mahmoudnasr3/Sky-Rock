// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract SkyRockLogger {
    struct DetectionEvent {
        uint256 timestamp;
        string sourceId;
        string className;
        uint256 confidence;
        string metadataHash;
    }

    DetectionEvent[] public detections;

    event DetectionLogged(
        uint256 indexed eventId,
        uint256 timestamp,
        string sourceId,
        string className,
        uint256 confidence,
        string metadataHash
    );

    function logDetection(
        string memory sourceId,
        string memory className,
        uint256 confidence,
        string memory metadataHash
    ) public {
        detections.push(
            DetectionEvent(
                block.timestamp,
                sourceId,
                className,
                confidence,
                metadataHash
            )
        );

        emit DetectionLogged(
            detections.length - 1,
            block.timestamp,
            sourceId,
            className,
            confidence,
            metadataHash
        );
    }

    function getDetectionCount() public view returns (uint256) {
        return detections.length;
    }
}